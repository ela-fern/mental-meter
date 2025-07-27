import os
from sqlalchemy import create_engine, Column, Integer, String, Date, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
import pandas as pd
import hashlib

# create missing tables
from sqlalchemy import create_engine
from .models import Base  # or wherever your models are

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to mental health entries
    entries = relationship("MentalHealthEntry", back_populates="user")

class MentalHealthEntry(Base):
    __tablename__ = "mental_health_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(Date, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to user
    user = relationship("User", back_populates="entries")
    
    # PHQ-9 symptoms (9 columns) - True: symptom present today, False: symptom not present
    phq9_0 = Column(Boolean, default=False)  # Little interest or pleasure
    phq9_1 = Column(Boolean, default=False)  # Feeling down, depressed
    phq9_2 = Column(Boolean, default=False)  # Sleep problems
    phq9_3 = Column(Boolean, default=False)  # Feeling tired
    phq9_4 = Column(Boolean, default=False)  # Poor appetite
    phq9_5 = Column(Boolean, default=False)  # Feeling bad about yourself
    phq9_6 = Column(Boolean, default=False)  # Trouble concentrating
    phq9_7 = Column(Boolean, default=False)  # Moving slowly or restless
    phq9_8 = Column(Boolean, default=False)  # Thoughts of death
    
    # GAD-7 symptoms (7 columns) - True: symptom present today, False: symptom not present
    gad7_0 = Column(Boolean, default=False)  # Feeling nervous
    gad7_1 = Column(Boolean, default=False)  # Can't stop worrying
    gad7_2 = Column(Boolean, default=False)  # Worrying too much
    gad7_3 = Column(Boolean, default=False)  # Trouble relaxing
    gad7_4 = Column(Boolean, default=False)  # Being restless
    gad7_5 = Column(Boolean, default=False)  # Easily annoyed
    gad7_6 = Column(Boolean, default=False)  # Feeling afraid

class DatabaseManager:
    def __init__(self, user_id=None):
        self.engine = engine
        self.SessionLocal = SessionLocal
        self.user_id = user_id
        self.create_tables()
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
    
    def hash_password(self, password):
        """Hash a password for storing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, password):
        """Create a new user and return success status"""
        session = self.get_session()
        try:
            # Check if user already exists
            existing_user = session.query(User).filter(User.username == username).first()
            if existing_user:
                return False
            
            # Create new user
            password_hash = self.hash_password(password)
            new_user = User(username=username, password_hash=password_hash)
            session.add(new_user)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            return False
        finally:
            session.close()
    
    def authenticate_user(self, username, password):
        """Authenticate a user and return user_id if successful"""
        session = self.get_session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if user and user.password_hash == self.hash_password(password):
                return user.id  # Return user_id directly
            return None  # Return None if authentication fails
        except Exception as e:
            return None
        finally:
            session.close()
    
    def set_user_id(self, user_id):
        """Set the current user ID"""
        self.user_id = user_id
    
    def save_daily_entry(self, date, phq9_responses, gad7_responses):
        """Save or update a daily entry for the current user"""
        if not self.user_id:
            raise ValueError("No user logged in")
        
        session = self.get_session()
        try:
            # Check if entry exists for this date and user
            existing_entry = session.query(MentalHealthEntry).filter(
                MentalHealthEntry.date == date,
                MentalHealthEntry.user_id == self.user_id
            ).first()
            
            if existing_entry:
                # Update existing entry
                # Update PHQ-9 responses
                for i in range(9):
                    key = f'phq9_{i}'
                    setattr(existing_entry, key, phq9_responses[i] if i < len(phq9_responses) else False)
                
                # Update GAD-7 responses
                for i in range(7):
                    key = f'gad7_{i}'
                    setattr(existing_entry, key, gad7_responses[i] if i < len(gad7_responses) else False)
                
                existing_entry.updated_at = datetime.utcnow()
                entry = existing_entry
            else:
                # Create new entry
                entry_data = {'date': date, 'user_id': self.user_id}
                
                # Add PHQ-9 responses
                for i in range(9):
                    key = f'phq9_{i}'
                    entry_data[key] = phq9_responses[i] if i < len(phq9_responses) else False
                
                # Add GAD-7 responses
                for i in range(7):
                    key = f'gad7_{i}'
                    entry_data[key] = gad7_responses[i] if i < len(gad7_responses) else False
                
                entry = MentalHealthEntry(**entry_data)
                session.add(entry)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            return False
        finally:
            session.close()
    
    def get_data_for_date(self, date):
        """Get data for a specific date for the current user"""
        if not self.user_id:
            return None
        
        session = self.get_session()
        try:
            entry = session.query(MentalHealthEntry).filter(
                MentalHealthEntry.date == date,
                MentalHealthEntry.user_id == self.user_id
            ).first()
            
            if not entry:
                return None
            
            # Convert to dictionary format matching the old CSV system
            result = {'date': entry.date.isoformat()}
            
            # Add PHQ-9 responses
            for i in range(9):
                key = f'phq9_{i}'
                result[key] = getattr(entry, key, False)
            
            # Add GAD-7 responses
            for i in range(7):
                key = f'gad7_{i}'
                result[key] = getattr(entry, key, False)
            
            return result
        
        finally:
            session.close()
    
    def get_entry_by_date(self, date):
        """Get entry object for a specific date for the current user"""
        if not self.user_id:
            return None
        
        session = self.get_session()
        try:
            entry = session.query(MentalHealthEntry).filter(
                MentalHealthEntry.date == date,
                MentalHealthEntry.user_id == self.user_id
            ).first()
            return entry
        finally:
            session.close()
    
    def get_historical_data(self, start_date=None, end_date=None):
        """Get historical data for a date range for the current user"""
        if not self.user_id:
            return []
        
        session = self.get_session()
        try:
            query = session.query(MentalHealthEntry).filter(
                MentalHealthEntry.user_id == self.user_id
            )
            
            if start_date:
                query = query.filter(MentalHealthEntry.date >= start_date)
            if end_date:
                query = query.filter(MentalHealthEntry.date <= end_date)
            
            entries = query.order_by(MentalHealthEntry.date).all()
            
            if not entries:
                return pd.DataFrame()
            
            # Convert to DataFrame format matching the old CSV system
            data = []
            for entry in entries:
                row = {'date': entry.date}
                
                # Add PHQ-9 responses
                for i in range(9):
                    key = f'phq9_{i}'
                    row[key] = getattr(entry, key, False)
                
                # Add GAD-7 responses
                for i in range(7):
                    key = f'gad7_{i}'
                    row[key] = getattr(entry, key, False)
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date')
        
        finally:
            session.close()
    
    def get_rolling_data(self, end_date, days=14):
        """Get data for rolling window calculation for the current user"""
        start_date = end_date - timedelta(days=days-1)
        return self.get_historical_data(start_date, end_date)
    
    def get_all_data(self):
        """Get all data for the current user"""
        if not self.user_id:
            return pd.DataFrame()
        
        session = self.get_session()
        try:
            entries = session.query(MentalHealthEntry).filter(
                MentalHealthEntry.user_id == self.user_id
            ).order_by(MentalHealthEntry.date).all()
            
            if not entries:
                return pd.DataFrame()
            
            # Convert to DataFrame format
            data = []
            for entry in entries:
                row = {'date': entry.date}
                
                # Add PHQ-9 responses
                for i in range(9):
                    key = f'phq9_{i}'
                    row[key] = getattr(entry, key, False)
                
                # Add GAD-7 responses
                for i in range(7):
                    key = f'gad7_{i}'
                    row[key] = getattr(entry, key, False)
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date')
        
        finally:
            session.close()
    
    def migrate_from_csv(self, csv_file="mental_health_data.csv"):
        """Migrate data from existing CSV file to database"""
        if not os.path.exists(csv_file):
            print(f"CSV file {csv_file} not found. No migration needed.")
            return
        
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                print("CSV file is empty. No migration needed.")
                return
            
            session = self.get_session()
            migrated_count = 0
            
            for _, row in df.iterrows():
                try:
                    date = pd.to_datetime(row['date']).date()
                    
                    # Check if entry already exists
                    existing = session.query(MentalHealthEntry).filter(
                        MentalHealthEntry.date == date
                    ).first()
                    
                    if existing:
                        continue  # Skip if already exists
                    
                    # Create new entry
                    entry_data = {'date': date}
                    
                    # Add PHQ-9 responses
                    for i in range(9):
                        key = f'phq9_{i}'
                        entry_data[key] = bool(row.get(key, False))
                    
                    # Add GAD-7 responses
                    for i in range(7):
                        key = f'gad7_{i}'
                        entry_data[key] = bool(row.get(key, False))
                    
                    entry = MentalHealthEntry(**entry_data)
                    session.add(entry)
                    migrated_count += 1
                
                except Exception as e:
                    print(f"Error migrating row: {e}")
                    continue
            
            session.commit()
            print(f"Successfully migrated {migrated_count} entries from CSV to database.")
            
            # Optionally backup the CSV file
            if migrated_count > 0:
                backup_name = f"{csv_file}.backup"
                os.rename(csv_file, backup_name)
                print(f"CSV file backed up as {backup_name}")
        
        except Exception as e:
            session.rollback()
            print(f"Error during migration: {e}")
        finally:
            session.close()
